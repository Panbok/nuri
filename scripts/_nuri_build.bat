@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

set "MODE=%~1"
set "PROFILE=%~2"
set "TRACY_MODE="
set "DEVCHECKS=ON"
set "FSR31_MODE=OFF"
if /I "%NURI_WITH_FSR31%"=="ON" set "FSR31_MODE=ON"

if "%MODE%"=="" goto usage
if "%PROFILE%"=="" goto usage
if /I not "%MODE%"=="debug" if /I not "%MODE%"=="release" goto usage

set "BUILD_APP=OFF"
set "BUILD_EDITOR=OFF"
set "BUILD_TESTS=OFF"
set "BUILD_TOOLS=OFF"
set "BUILD_BENCHMARK_CLI=ON"
set "BUILD_SNAPSHOT_CLI=ON"
set "BUILD_SNAPSHOT_TESTING=OFF"
set "BUILD_AUTOTEST_CLI=OFF"
set "BUILD_AUTOTESTING=OFF"
set "BUILD_TARGET="
set "BUILD_DIR_SUFFIX=%MODE%"
set "MANIFEST_FEATURES=%VCPKG_MANIFEST_FEATURES%"

if /I "%PROFILE%"=="lib" (
  set "BUILD_TARGET=nuri_renderer"
) else if /I "%PROFILE%"=="app" (
  set "BUILD_APP=ON"
  set "BUILD_TARGET=nuri"
) else if /I "%PROFILE%"=="editor" (
  set "BUILD_EDITOR=ON"
  set "BUILD_TARGET=nuri_editor"
  call :append_manifest_feature editor
) else if /I "%PROFILE%"=="tests" (
  set "BUILD_TESTS=ON"
  call :append_manifest_feature tests
) else if /I "%PROFILE%"=="bench" (
  set "BUILD_TOOLS=ON"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_TARGET=nuri-bench"
  set "BUILD_DIR_SUFFIX=%MODE%-bench"
  set "DEVCHECKS=OFF"
  call :append_manifest_feature benchmark-tools
) else if /I "%PROFILE%"=="benchmark" (
  set "PROFILE=bench"
  set "BUILD_TOOLS=ON"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_TARGET=nuri-bench"
  set "BUILD_DIR_SUFFIX=%MODE%-bench"
  set "DEVCHECKS=OFF"
  call :append_manifest_feature benchmark-tools
) else if /I "%PROFILE%"=="snapshot" (
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_TARGET=nuri-snapshot"
  set "BUILD_DIR_SUFFIX=%MODE%-snapshot"
  set "DEVCHECKS=OFF"
  call :append_manifest_feature snapshot-tools
) else if /I "%PROFILE%"=="autotest" (
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_AUTOTEST_CLI=ON"
  set "BUILD_AUTOTESTING=ON"
  set "BUILD_TARGET=nuri-autotest"
  set "BUILD_DIR_SUFFIX=%MODE%-autotest"
  set "DEVCHECKS=OFF"
  call :append_manifest_feature autotest-tools
) else if /I "%PROFILE%"=="autotests" (
  set "PROFILE=autotest"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_AUTOTEST_CLI=ON"
  set "BUILD_AUTOTESTING=ON"
  set "BUILD_TARGET=nuri-autotest"
  set "BUILD_DIR_SUFFIX=%MODE%-autotest"
  set "DEVCHECKS=OFF"
  call :append_manifest_feature autotest-tools
) else if /I "%PROFILE%"=="snapshots" (
  set "PROFILE=snapshot"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_TARGET=nuri-snapshot"
  set "BUILD_DIR_SUFFIX=%MODE%-snapshot"
  set "DEVCHECKS=OFF"
  call :append_manifest_feature snapshot-tools
) else if /I "%PROFILE%"=="bench-tests" (
  set "BUILD_TESTS=ON"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_DIR_SUFFIX=%MODE%-bench-tests"
  call :append_manifest_feature benchmark-tools
  call :append_manifest_feature tests
) else if /I "%PROFILE%"=="benchmark-tests" (
  set "PROFILE=bench-tests"
  set "BUILD_TESTS=ON"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_DIR_SUFFIX=%MODE%-bench-tests"
  call :append_manifest_feature benchmark-tools
  call :append_manifest_feature tests
) else if /I "%PROFILE%"=="snapshot-tests" (
  set "BUILD_TESTS=ON"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_DIR_SUFFIX=%MODE%-snapshot-tests"
  call :append_manifest_feature snapshot-tools
  call :append_manifest_feature tests
) else if /I "%PROFILE%"=="autotest-tests" (
  set "BUILD_TESTS=ON"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_AUTOTEST_CLI=OFF"
  set "BUILD_AUTOTESTING=ON"
  set "BUILD_DIR_SUFFIX=%MODE%-autotest-tests"
  call :append_manifest_feature autotest-tools
  call :append_manifest_feature tests
) else if /I "%PROFILE%"=="autotests-tests" (
  set "PROFILE=autotest-tests"
  set "BUILD_TESTS=ON"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_AUTOTEST_CLI=OFF"
  set "BUILD_AUTOTESTING=ON"
  set "BUILD_DIR_SUFFIX=%MODE%-autotest-tests"
  call :append_manifest_feature autotest-tools
  call :append_manifest_feature tests
) else if /I "%PROFILE%"=="snapshots-tests" (
  set "PROFILE=snapshot-tests"
  set "BUILD_TESTS=ON"
  set "BUILD_TOOLS=ON"
  set "BUILD_BENCHMARK_CLI=OFF"
  set "BUILD_SNAPSHOT_CLI=OFF"
  set "BUILD_SNAPSHOT_TESTING=ON"
  set "BUILD_DIR_SUFFIX=%MODE%-snapshot-tests"
  call :append_manifest_feature snapshot-tools
  call :append_manifest_feature tests
) else (
  goto usage
)

shift
shift
:parse_args
if "%~1"=="" goto parsed_args
if /I "%~1"=="cpu" (
  if not "%TRACY_MODE%"=="" goto usage
  set "TRACY_MODE=cpu"
) else if /I "%~1"=="off" (
  if not "%TRACY_MODE%"=="" goto usage
  set "TRACY_MODE=off"
) else if /I "%~1"=="devchecks" (
  set "DEVCHECKS=ON"
) else (
  goto usage
)
shift
goto parse_args

:parsed_args
set "NURI_WITH_ASAN=OFF"
set "NURI_WITH_LOGGING=OFF"
set "NURI_WITH_ASSERTS=OFF"
set "NURI_WITH_TRACY=OFF"

if /I "%MODE%"=="debug" (
  set "NURI_WITH_ASAN=ON"
  set "NURI_WITH_LOGGING=ON"
  set "NURI_WITH_ASSERTS=ON"
  set "NURI_WITH_TRACY=ON"
)

if /I "%MODE%"=="release" if /I "%DEVCHECKS%"=="ON" (
  set "NURI_WITH_LOGGING=ON"
  set "NURI_WITH_ASSERTS=ON"
)

if /I "%TRACY_MODE%"=="cpu" (
  set "NURI_WITH_TRACY=ON"
) else if /I "%TRACY_MODE%"=="off" (
  set "NURI_WITH_TRACY=OFF"
) else if not "%TRACY_MODE%"=="" (
  goto usage
)

if "%VCPKG_ROOT%"=="" (
  echo VCPKG_ROOT is not set. Point it at your vcpkg root.
  exit /b 1
)
set "NURI_VCPKG_ROOT=%VCPKG_ROOT%"

set "GENERATOR=Ninja"
set "GENERATOR_ARGS="
set "C_COMPILER_ARG="
set "CXX_COMPILER_ARG="
set "LINKER_ARG="
set "EXPECTED_CXX_COMPILER="
set "EXPECTED_LINKER="

set "NINJA_EXE="
for /f "delims=" %%i in ('where ninja 2^>nul') do (
  set "NINJA_EXE=%%i"
  goto have_ninja
)
:have_ninja
if not defined NINJA_EXE (
  echo Ninja not found. Install ninja or run CMake with a different generator.
  exit /b 1
)

set "VSDEVCMD="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "delims=" %%i in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do (
    if exist "%%i\Common7\Tools\VsDevCmd.bat" set "VSDEVCMD=%%i\Common7\Tools\VsDevCmd.bat"
  )
)
if not defined VSDEVCMD (
  for %%v in (18 2022 2019) do (
    for %%e in (BuildTools Community Professional Enterprise) do (
      if not defined VSDEVCMD if exist "%ProgramFiles%\Microsoft Visual Studio\%%v\%%e\Common7\Tools\VsDevCmd.bat" set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\%%v\%%e\Common7\Tools\VsDevCmd.bat"
      if not defined VSDEVCMD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\%%v\%%e\Common7\Tools\VsDevCmd.bat" set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\%%v\%%e\Common7\Tools\VsDevCmd.bat"
    )
  )
)
if not defined VSDEVCMD (
  echo Visual Studio developer command script not found. Install the C++ Build Tools workload.
  exit /b 1
)
call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
set "VCPKG_ROOT=%NURI_VCPKG_ROOT%"

if /I "%MODE%"=="debug" goto set_debug_toolchain
if /I "%MODE%"=="release" goto set_release_toolchain
goto usage

:set_debug_toolchain
set "C_COMPILER_ARG=-DCMAKE_C_COMPILER=clang"
set "CXX_COMPILER_ARG=-DCMAKE_CXX_COMPILER=clang++"
set "LINKER_ARG="
set "EXPECTED_CXX_COMPILER=clang++"
set "EXPECTED_LINKER=lld-link"
goto toolchain_ready

:set_release_toolchain
set "C_COMPILER_ARG=-DCMAKE_C_COMPILER=clang-cl"
set "CXX_COMPILER_ARG=-DCMAKE_CXX_COMPILER=clang-cl"
set "LINKER_ARG=-DCMAKE_LINKER=link.exe"
set "EXPECTED_CXX_COMPILER=clang-cl"
set "EXPECTED_LINKER=link.exe"

:toolchain_ready

for %%i in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fi"
set "BUILD_DIR=%REPO_ROOT%\build\%BUILD_DIR_SUFFIX%"
set "CONFIG_STAMP_FILE=%BUILD_DIR%\.nuri_config.txt"
set "TOOLCHAIN_STAMP_FILE=%BUILD_DIR%\.nuri_toolchain.txt"
set "DESIRED_CONFIG=mode=%MODE% profile=%PROFILE% target=%BUILD_TARGET% app=%BUILD_APP% editor=%BUILD_EDITOR% tests=%BUILD_TESTS% tools=%BUILD_TOOLS% benchmark_cli=%BUILD_BENCHMARK_CLI% snapshot_cli=%BUILD_SNAPSHOT_CLI% snapshot_testing=%BUILD_SNAPSHOT_TESTING% autotest_cli=%BUILD_AUTOTEST_CLI% autotesting=%BUILD_AUTOTESTING% features=%MANIFEST_FEATURES% tracy=%TRACY_MODE% devchecks=%DEVCHECKS% fsr31=%FSR31_MODE% cc=%C_COMPILER_ARG% cxx=%CXX_COMPILER_ARG% linker=%LINKER_ARG% asan=%NURI_WITH_ASAN% logging=%NURI_WITH_LOGGING% asserts=%NURI_WITH_ASSERTS% tracy_cpu=%NURI_WITH_TRACY%"
set "TOOLCHAIN_SIGNATURE=vs=%VisualStudioVersion% vc=%VCToolsVersion% sdk=%WindowsSDKVersion% cxx=%EXPECTED_CXX_COMPILER% linker=%EXPECTED_LINKER%"
set "TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
set "MAKE_PROGRAM_ARG=-DCMAKE_MAKE_PROGRAM=%NINJA_EXE%"
set "VCPKG_EXECUTABLE_ARG=-DVCPKG_EXECUTABLE=%VCPKG_ROOT%\vcpkg.exe"
set "MANIFEST_FEATURES_ARG="
if not "%MANIFEST_FEATURES%"=="" set "MANIFEST_FEATURES_ARG=-DVCPKG_MANIFEST_FEATURES=%MANIFEST_FEATURES%"

set "SHOULD_CONFIGURE=1"
set "SHOULD_CLEAN=0"
if exist "%BUILD_DIR%\CMakeCache.txt" if exist "%CONFIG_STAMP_FILE%" (
  set /p "EXISTING_CONFIG=" < "%CONFIG_STAMP_FILE%"
  if "!EXISTING_CONFIG!"=="%DESIRED_CONFIG%" set "SHOULD_CONFIGURE=0"
)
if exist "%BUILD_DIR%\CMakeCache.txt" (
  if not exist "%TOOLCHAIN_STAMP_FILE%" (
    set "SHOULD_CLEAN=1"
    set "SHOULD_CONFIGURE=1"
  ) else (
    set /p "EXISTING_TOOLCHAIN=" < "%TOOLCHAIN_STAMP_FILE%"
    if /I not "!EXISTING_TOOLCHAIN!"=="%TOOLCHAIN_SIGNATURE%" (
      set "SHOULD_CLEAN=1"
      set "SHOULD_CONFIGURE=1"
    )
  )
)

if "%SHOULD_CONFIGURE%"=="0" goto build_target
if "%SHOULD_CLEAN%"=="1" if exist "%BUILD_DIR%" (
  echo Toolchain changed; removing stale build tree "%BUILD_DIR%".
  rmdir /s /q "%BUILD_DIR%"
)
if /I "%MODE%"=="debug" goto configure_debug
if /I "%MODE%"=="release" goto configure_release
goto usage

:configure_debug
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -G "%GENERATOR%" ^
  -DCMAKE_BUILD_TYPE=Debug ^
  %C_COMPILER_ARG% ^
  %CXX_COMPILER_ARG% ^
  %LINKER_ARG% ^
  %MAKE_PROGRAM_ARG% ^
  %VCPKG_EXECUTABLE_ARG% ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DVCPKG_APPLOCAL_DEPS=OFF ^
  %MANIFEST_FEATURES_ARG% ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_TOOL_PROFILE="%PROFILE%" ^
  -DNURI_DEV_CHECKS="%DEVCHECKS%" ^
  -DNURI_BUILD_APP="%BUILD_APP%" ^
  -DNURI_BUILD_EDITOR="%BUILD_EDITOR%" ^
  -DNURI_BUILD_TESTS="%BUILD_TESTS%" ^
  -DNURI_BUILD_TOOLS="%BUILD_TOOLS%" ^
  -DNURI_BUILD_BENCHMARK_CLI="%BUILD_BENCHMARK_CLI%" ^
  -DNURI_BUILD_SNAPSHOT_CLI="%BUILD_SNAPSHOT_CLI%" ^
  -DNURI_BUILD_SNAPSHOT_TESTING="%BUILD_SNAPSHOT_TESTING%" ^
  -DNURI_BUILD_AUTOTEST_CLI="%BUILD_AUTOTEST_CLI%" ^
  -DNURI_BUILD_AUTOTESTING="%BUILD_AUTOTESTING%" ^
  -DNURI_BUILD_SHARED=ON ^
  -DNURI_WITH_ASAN="%NURI_WITH_ASAN%" ^
  -DNURI_WITH_LOGGING="%NURI_WITH_LOGGING%" ^
  -DNURI_WITH_ASSERTS="%NURI_WITH_ASSERTS%" ^
  -DNURI_WITH_FSR31="%FSR31_MODE%" ^
  -DNURI_WITH_TRACY="%NURI_WITH_TRACY%"
if errorlevel 1 exit /b 1
goto write_config_stamp

:configure_release
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -G "%GENERATOR%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  %C_COMPILER_ARG% ^
  %CXX_COMPILER_ARG% ^
  %LINKER_ARG% ^
  %MAKE_PROGRAM_ARG% ^
  %VCPKG_EXECUTABLE_ARG% ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-static ^
  %MANIFEST_FEATURES_ARG% ^
  -DVCPKG_BUILD_TYPE=release ^
  -DNURI_TOOL_PROFILE="%PROFILE%" ^
  -DNURI_DEV_CHECKS="%DEVCHECKS%" ^
  -DNURI_BUILD_APP="%BUILD_APP%" ^
  -DNURI_BUILD_EDITOR="%BUILD_EDITOR%" ^
  -DNURI_BUILD_TESTS="%BUILD_TESTS%" ^
  -DNURI_BUILD_TOOLS="%BUILD_TOOLS%" ^
  -DNURI_BUILD_BENCHMARK_CLI="%BUILD_BENCHMARK_CLI%" ^
  -DNURI_BUILD_SNAPSHOT_CLI="%BUILD_SNAPSHOT_CLI%" ^
  -DNURI_BUILD_SNAPSHOT_TESTING="%BUILD_SNAPSHOT_TESTING%" ^
  -DNURI_BUILD_AUTOTEST_CLI="%BUILD_AUTOTEST_CLI%" ^
  -DNURI_BUILD_AUTOTESTING="%BUILD_AUTOTESTING%" ^
  -DNURI_BUILD_SHARED=OFF ^
  -DNURI_WITH_ASAN="%NURI_WITH_ASAN%" ^
  -DNURI_WITH_LOGGING="%NURI_WITH_LOGGING%" ^
  -DNURI_WITH_ASSERTS="%NURI_WITH_ASSERTS%" ^
  -DNURI_WITH_FSR31="%FSR31_MODE%" ^
  -DNURI_WITH_TRACY="%NURI_WITH_TRACY%"
if errorlevel 1 exit /b 1
goto write_config_stamp

:write_config_stamp
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
> "%CONFIG_STAMP_FILE%" echo(%DESIRED_CONFIG%
> "%TOOLCHAIN_STAMP_FILE%" echo(%TOOLCHAIN_SIGNATURE%

:build_target
if "%BUILD_TARGET%"=="" (
  cmake --build "%BUILD_DIR%"
  set "BUILD_EXIT=!errorlevel!"
) else (
  cmake --build "%BUILD_DIR%" --target "%BUILD_TARGET%"
  set "BUILD_EXIT=!errorlevel!"
)
exit /b !BUILD_EXIT!

:append_manifest_feature
set "FEATURE_TO_APPEND=%~1"
if "%FEATURE_TO_APPEND%"=="" exit /b 0
if "%MANIFEST_FEATURES%"=="" (
  set "MANIFEST_FEATURES=%FEATURE_TO_APPEND%"
) else (
  echo ;!MANIFEST_FEATURES!; | findstr /I /C:";%FEATURE_TO_APPEND%;" >nul
  if errorlevel 1 set "MANIFEST_FEATURES=!MANIFEST_FEATURES!;%FEATURE_TO_APPEND%"
)
exit /b 0

:usage
echo Usage: %~nx0 ^<debug^|release^> ^<lib^|app^|editor^|tests^|bench^|bench-tests^|snapshot^|snapshot-tests^|autotest^|autotest-tests^> [cpu^|off] [devchecks]
exit /b 1

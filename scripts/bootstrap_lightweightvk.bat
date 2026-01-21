@echo off
setlocal

set "ROOT=%~dp0.."
set "LVK_DIR=%ROOT%\\external\\lightweightvk"

set "HAS_GIT=0"
where git >nul 2>nul
if %errorlevel% equ 0 set "HAS_GIT=1"

set "PYTHON="
where py >nul 2>nul
if %errorlevel% equ 0 set "PYTHON=py -3"
if "%PYTHON%"=="" (
  where python3 >nul 2>nul
  if %errorlevel% equ 0 set "PYTHON=python3"
)
if "%PYTHON%"=="" (
  where python >nul 2>nul
  if %errorlevel% equ 0 set "PYTHON=python"
)
if "%PYTHON%"=="" (
  echo Python 3 not found on PATH. Install Python 3 ^(or the Windows py launcher^) and restart the terminal.
  exit /b 1
)

%PYTHON% -c "import sys; raise SystemExit(0 if sys.version_info[0]==3 else 1)" >nul 2>nul
if %errorlevel% neq 0 (
  echo Found Python, but it is not Python 3. Ensure `py -3` or `python3` points to Python 3.
  exit /b 1
)

rem If lightweightvk is configured as a real git submodule (gitlink entry), update it.
rem If the folder exists but isn't a registered submodule, just use it as a plain checkout.
if not exist "%LVK_DIR%\\CMakeLists.txt" (
  if "%HAS_GIT%"=="1" (
    if exist "%ROOT%\\.gitmodules" (
      set "LVK_PATH="
      if exist "%ROOT%\\external\\lightweightvk\\.git" set "LVK_PATH=external/lightweightvk"

      if "%LVK_PATH%"=="" (
        findstr /C:"external/lightweightvk" "%ROOT%\\.gitmodules" >nul 2>nul && set "LVK_PATH=external/lightweightvk"
      )

      if "%LVK_PATH%"=="" set "LVK_PATH=external/lightweightvk"

      git -C "%ROOT%" ls-files --stage "%LVK_PATH%" 2>nul | findstr /B "160000 " >nul 2>nul
      if %errorlevel% equ 0 (
        git -C "%ROOT%" submodule update --init --recursive "%LVK_PATH%"
        if errorlevel 1 exit /b 1
      )
    )
  )
)

if not exist "%LVK_DIR%\\CMakeLists.txt" (
  if "%HAS_GIT%"=="0" (
    echo git not found on PATH and lightweightvk is missing at: %LVK_DIR%
    echo Install Git for Windows or manually clone https://github.com/corporateshark/lightweightvk into that folder.
    exit /b 1
  )
  echo lightweightvk submodule not found at: %LVK_DIR%
  exit /b 1
)

pushd "%LVK_DIR%"
if /I "%~1"=="--full" (
  %PYTHON% deploy_content.py
  if errorlevel 1 (
    popd
    exit /b 1
  )
  %PYTHON% deploy_deps.py
  if errorlevel 1 (
    popd
    exit /b 1
  )
) else (
  %PYTHON% third-party\bootstrap.py -b third-party\deps --bootstrap-file=third-party\bootstrap-deps.json -N "%~dp0lvk-deps.txt" --break-on-first-error
)
if errorlevel 1 (
  popd
  exit /b 1
)
popd

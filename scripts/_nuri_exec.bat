@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"
set "MODE=%~1"
set "PROFILE=%~2"
if "%MODE%"=="" exit /b 2
if "%PROFILE%"=="" exit /b 2
shift
shift
python "%SCRIPT_DIR%nuri_build.py" legacy-run "%MODE%" "%PROFILE%" --no-build -- %*
exit /b %errorlevel%

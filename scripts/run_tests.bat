@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"
python "%SCRIPT_DIR%nuri_build.py" legacy-wrapper-test %*
exit /b %errorlevel%

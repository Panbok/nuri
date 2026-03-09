@echo off
setlocal
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

call "%SCRIPT_DIR%run_app.bat" debug %*
exit /b %errorlevel%

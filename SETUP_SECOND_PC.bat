@echo off
setlocal
title SpeechMind - Complete Windows Setup
cd /d "%~dp0"

echo ==========================================================
echo SPEECHMIND - COMPLETE WINDOWS SETUP
echo ==========================================================
echo Running the PowerShell installer...
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0SETUP_SECOND_PC.ps1"
set "SETUP_EXIT=%ERRORLEVEL%"

echo.
if not "%SETUP_EXIT%"=="0" goto FAILED

echo Setup completed successfully.
goto END

:FAILED
echo Setup failed with exit code %SETUP_EXIT%.

:END
echo.
echo Log: %~dp0second_pc_setup.log
echo Press any key to close this window.
pause >nul
exit /b %SETUP_EXIT%

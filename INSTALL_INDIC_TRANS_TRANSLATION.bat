@echo off
setlocal
cd /d "%~dp0"

echo IndicTransToolkit is not supported on native Windows.
echo SpeechMind now uses an isolated WSL Linux translation worker.
echo.
call "%~dp0SETUP_INDIC_TRANS_WSL.bat"
exit /b %ERRORLEVEL%

@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_SPEECHMIND_BODHAN.ps1"
if errorlevel 1 (
    echo.
    echo SpeechMind did not start successfully. Review the error above.
    pause
)

@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo SpeechMind main environment not found. Run SETUP_SECOND_PC.bat first.
    pause
    exit /b 1
)
if not exist ".venv_srota\Scripts\python.exe" (
    echo Srota environment not found. Run SETUP_HINGLISH_ASR_ENV.bat once.
    pause
    exit /b 1
)

echo ==========================================================
echo SpeechMind - Srota integration check
echo ==========================================================
echo 1. Runs the lightweight Srota provider protocol tests.
echo 2. Runs one prepared Hinglish clip through automatic SpeechMind ASR.
echo No dataset download, model training, or full benchmark is performed.
echo.

".venv\Scripts\python.exe" -m unittest discover -s tests -p "test_srota_provider.py"
if errorlevel 1 goto failed

".venv\Scripts\python.exe" "%~dp0scripts\test_srota_auto_integration.py"
if errorlevel 1 goto failed

echo.
echo Srota integration check passed.
pause
exit /b 0

:failed
echo.
echo Srota integration check failed. Keep the console output for debugging.
pause
exit /b 1

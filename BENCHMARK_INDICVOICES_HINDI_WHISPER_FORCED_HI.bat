@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Whisper Small forced-Hindi ASR benchmark
echo Diagnostic comparison only: language=hi is forced internally.
echo This is NOT a language selector in the SpeechMind user interface.
echo Local inference only. No paid API is used.
echo.
"%PYTHON_EXE%" "%~dp0scripts\evaluate_indicvoices_asr.py" --model whisper-hi
if errorlevel 1 (
    echo.
    echo Forced-Hindi Whisper benchmark failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

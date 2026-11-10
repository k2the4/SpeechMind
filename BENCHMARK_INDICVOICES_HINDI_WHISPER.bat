@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Forced Whisper Small Hindi ASR benchmark
echo Runs Whisper Small on every Hindi validation clip.
echo Local inference only. No paid API is used.
echo This run is for comparison with the same IndicConformer sample.
echo.
"%PYTHON_EXE%" "%~dp0scripts\evaluate_indicvoices_asr.py" --model whisper
if errorlevel 1 (
    echo.
    echo Whisper benchmark failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

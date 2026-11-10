@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Local Hindi ASR benchmark
echo Default model=auto (current local ASR routing).
echo This may take some time on CPU.
echo.
"%PYTHON_EXE%" "%~dp0scripts\evaluate_indicvoices_asr.py" --model auto
if errorlevel 1 (
    echo.
    echo Benchmark failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

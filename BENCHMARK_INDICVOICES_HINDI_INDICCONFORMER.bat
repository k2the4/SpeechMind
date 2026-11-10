@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Forced IndicConformer Hindi ASR benchmark
echo Runs IndicConformer on every Hindi validation clip.
echo Local inference only. No paid API is used.
echo Model initialization is reported separately from clip timing.
echo.
"%PYTHON_EXE%" "%~dp0scripts\evaluate_indicvoices_asr.py" --model indicconformer
if errorlevel 1 (
    echo.
    echo IndicConformer benchmark failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

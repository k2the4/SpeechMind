@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - IndicVoices Hindi access and schema check
echo This reads at most 3 streamed validation records and saves no audio.
echo.
"%PYTHON_EXE%" "%~dp0scripts\inspect_indicvoices.py"
if errorlevel 1 (
    echo.
    echo Inspection did not complete. Review the error above.
    pause
    exit /b 1
)

echo.
pause

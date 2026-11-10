@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Inspect existing extracted HiACC folder
echo This does NOT download or extract the corpus again.
echo It reports audio folders, transcript files, split names and metadata.
echo.
"%PYTHON_EXE%" "%~dp0scripts\inspect_hiacc_layout.py"
if errorlevel 1 (
    echo.
    echo HiACC inventory failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

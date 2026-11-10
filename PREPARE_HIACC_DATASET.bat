@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Prepare public HiACC Hinglish corpus
echo Dataset archive: approximately 531.6 MB.
echo The script verifies the Zenodo MD5 before extraction.
echo Files stay in data\hiacc and are not added to Git.
echo.
"%PYTHON_EXE%" "%~dp0scripts\prepare_hiacc_dataset.py"
if errorlevel 1 (
    echo.
    echo HiACC preparation failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

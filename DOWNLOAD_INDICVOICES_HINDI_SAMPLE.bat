@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo SpeechMind - Download a small Hindi IndicVoices validation sample
echo This defaults to 100 clips from 100 different speakers.
echo It streams data and saves only selected FLAC files.
echo No paid API or full-dataset download is used.
echo.
"%PYTHON_EXE%" "%~dp0scripts\download_indicvoices_sample.py" --max-records 100 --max-per-speaker 1
if errorlevel 1 (
    echo.
    echo Download failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

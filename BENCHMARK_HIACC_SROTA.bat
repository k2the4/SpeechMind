@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv_srota\Scripts\python.exe" (
    echo Srota environment not found. Run SETUP_HINGLISH_ASR_ENV.bat first.
    pause
    exit /b 1
)

echo SpeechMind - Srota Hinglish ASR smoke benchmark
echo 20 test utterances, balanced across adults/children where possible.
echo Local CPU inference only. No paid API.
echo The first run downloads model weights from Hugging Face.
echo.
".venv_srota\Scripts\python.exe" "%~dp0scripts\benchmark_srota_hiacc.py" --limit 20 --max-per-speaker 1
if errorlevel 1 (
    echo.
    echo Srota benchmark failed. Existing SpeechMind .venv is untouched.
    pause
    exit /b 1
)
echo.
pause

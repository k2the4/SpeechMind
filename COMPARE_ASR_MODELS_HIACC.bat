@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Main SpeechMind virtual environment not found.
    echo Expected .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist ".venv_srota\Scripts\python.exe" (
    echo Srota environment not found.
    echo Run SETUP_HINGLISH_ASR_ENV.bat first.
    pause
    exit /b 1
)

set SPEECHMIND_SROTA_TORCH_THREADS=4

".venv\Scripts\python.exe" scripts\benchmark_asr_models_paired.py --limit 20 --seed 42 --max-per-speaker 1

if errorlevel 1 (
    echo.
    echo Paired ASR benchmark failed. Review the error above.
    pause
    exit /b 1
)

echo.
echo Paired ASR benchmark finished.
echo See data\hiacc\benchmarks\paired_asr\summary.json
echo and predictions.csv.
pause

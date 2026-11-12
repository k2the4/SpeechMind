@echo off
setlocal
cd /d "%~dp0"

if not exist "data\hiacc\manifest.csv" (
    echo HiACC manifest not found.
    echo Run PREPARE_HIACC_DATASET.bat first.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Main SpeechMind virtual environment not found.
    echo Expected .venv\Scripts\python.exe
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\evaluate_speechmind_manifest.py ^
  --input "data\hiacc\manifest.csv" ^
  --audio-root "data\hiacc" ^
  --split test ^
  --limit 40 ^
  --seed 42 ^
  --max-per-speaker 1

if errorlevel 1 (
    echo.
    echo Evaluation failed. Review the error above.
    pause
    exit /b 1
)

echo.
echo Evaluation finished. See data\benchmarks\full_pipeline for summary.json and predictions.csv.
pause

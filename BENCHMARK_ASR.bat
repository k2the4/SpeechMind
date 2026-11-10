@echo off
setlocal
cd /d "%~dp0"

if not exist "data\asr_benchmark\manifest.csv" (
    echo Missing data\asr_benchmark\manifest.csv
    echo Create it from docs\asr_manifest_template.csv and add audio_path,reference rows.
    echo Request the IITG-HingCoS dataset using the official link in docs\DATA_SOURCES.md.
    pause
    exit /b 1
)

echo Comparing local Whisper and Bodhan ASR...
python -m scripts.benchmark_asr --backend all --manifest data/asr_benchmark/manifest.csv
if errorlevel 1 (
    echo ASR benchmark failed. Review the error above.
    pause
    exit /b 1
)
echo.
echo Results saved under training_outputs\asr_benchmark
pause

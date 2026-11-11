@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo SpeechMind main environment not found.
    pause
    exit /b 1
)
if not exist "data\hiacc\benchmarks\srota\predictions.csv" (
    echo Srota predictions not found. Run BENCHMARK_HIACC_SROTA.bat first.
    pause
    exit /b 1
)

echo SpeechMind - Current automatic ASR route on the exact 20 HiACC clips
echo Uses Whisper language detection and the existing IndicConformer/Whisper routing.
echo No dependency installation. No production routing changes.
echo.
".venv\Scripts\python.exe" "%~dp0scripts\benchmark_hiacc_speechmind_auto.py"
if errorlevel 1 (
    echo.
    echo Automatic ASR benchmark failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo SpeechMind main environment not found. Run SETUP_SECOND_PC.bat only if this PC has not been set up.
    pause
    exit /b 1
)
if not exist "data\hiacc\benchmarks\srota\predictions.csv" (
    echo Srota predictions not found.
    echo Run BENCHMARK_HIACC_SROTA.bat first.
    pause
    exit /b 1
)

echo SpeechMind - IndicConformer diagnostic on the exact Srota HiACC clips
echo This is a forced-Hindi cross-language diagnostic, not a Hinglish specialist benchmark.
echo Uses the existing .venv. No dependency installation.
echo.
".venv\Scripts\python.exe" "%~dp0scripts\compare_hiacc_indicconformer.py"
if errorlevel 1 (
    echo.
    echo IndicConformer comparison failed. Review the error above.
    pause
    exit /b 1
)
echo.
pause

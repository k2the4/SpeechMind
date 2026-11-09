@echo off
setlocal EnableExtensions
title SpeechMind - WavLM Voice Emotion Training
cd /d "%~dp0"

set "LOG_FILE=%~dp0training_launcher.log"
set "EXIT_CODE=0"

> "%LOG_FILE%" echo SpeechMind WavLM training launcher
>> "%LOG_FILE%" echo Started: %DATE% %TIME%
>> "%LOG_FILE%" echo Project folder: %CD%

echo ==========================================
echo SpeechMind - WavLM voice emotion training
echo ==========================================
echo.
echo This experiment uses WavLM with full fine-tuning.
echo It may take several hours on a CPU.
echo The previous checkpoint will NOT be overwritten.
echo A log will be saved as training_launcher.log.
echo.

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Project virtual environment not found.
    echo Expected: %PYTHON_EXE%
    >> "%LOG_FILE%" echo ERROR: Virtual environment not found.
    set "EXIT_CODE=2"
    goto FINISH
)

echo Checking Python and PyTorch...
"%PYTHON_EXE%" --version >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ERROR: Python could not start. See training_launcher.log.
    set "EXIT_CODE=3"
    goto FINISH
)
"%PYTHON_EXE%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ERROR: PyTorch is not working in this environment.
    set "EXIT_CODE=4"
    goto FINISH
)

"%PYTHON_EXE%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: CUDA is not available. Full WavLM training may take many hours.
    echo Keep the computer plugged in and ensure enough free disk space.
    echo.
    choice /C YN /N /M "Continue with CPU training? [Y/N] "
    if errorlevel 2 (
        echo Training cancelled by user.
        >> "%LOG_FILE%" echo Training cancelled by user.
        goto FINISH
    )
)

echo.
echo Starting 12-epoch maximum WavLM fine-tuning.
echo Early stopping uses validation macro-F1.
echo The test split is held out for final evaluation.
echo.

>> "%LOG_FILE%" echo Starting WavLM training at %DATE% %TIME%
"%PYTHON_EXE%" "%~dp0scripts\train_voice_emotion.py" --epochs 12 --batch-size 2 --backbone-learning-rate 1e-5 --head-learning-rate 5e-5 >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Training and held-out test evaluation completed.
    echo Report: training_outputs\wavlm_ravdess\test_metrics.json
    echo New checkpoint: checkpoints\voice_emotion_wavlm
    >> "%LOG_FILE%" echo Training completed successfully.
) else (
    echo.
    echo Training failed with exit code %EXIT_CODE%.
    echo Open training_launcher.log to see the full error.
    >> "%LOG_FILE%" echo Training failed with exit code %EXIT_CODE%.
)

:FINISH
echo.
echo ==========================================
echo Launcher finished. Exit code: %EXIT_CODE%
echo Log file: %LOG_FILE%
echo ==========================================
echo.
pause
exit /b %EXIT_CODE%

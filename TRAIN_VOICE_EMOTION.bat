@echo off
setlocal EnableExtensions
title SpeechMind - Voice Emotion Training
cd /d "%~dp0"

set "LOG_FILE=%~dp0training_launcher.log"
set "EXIT_CODE=0"

> "%LOG_FILE%" echo SpeechMind voice-emotion training launcher
>> "%LOG_FILE%" echo Started: %DATE% %TIME%
>> "%LOG_FILE%" echo Project folder: %CD%

echo ==========================================
echo SpeechMind - Voice emotion fine-tuning
echo ==========================================
echo.
echo This launcher checks your local Python environment,
echo then asks before starting CPU training.
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
>> "%LOG_FILE%" echo Checking Python and PyTorch...
"%PYTHON_EXE%" --version >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ERROR: Python could not start. See training_launcher.log.
    >> "%LOG_FILE%" echo ERROR: Python could not start.
    set "EXIT_CODE=3"
    goto FINISH
)

"%PYTHON_EXE%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ERROR: PyTorch is not working in this environment.
    echo See training_launcher.log.
    set "EXIT_CODE=4"
    goto FINISH
)

"%PYTHON_EXE%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: CUDA is not available. Training may take many hours on CPU.
    echo Keep the computer plugged in. Do not start if you cannot leave it running.
    echo.
    choice /C YN /N /M "Continue with CPU training? [Y/N] "
    if errorlevel 2 (
        echo Training cancelled by user.
        >> "%LOG_FILE%" echo Training cancelled by user.
        set "EXIT_CODE=0"
        goto FINISH
    )
)

echo.
echo Starting voice-emotion training...
echo This may take a long time. Do not close this window.
echo.
>> "%LOG_FILE%" echo Starting training at %DATE% %TIME%

"%PYTHON_EXE%" "%~dp0scripts\train_voice_emotion.py" --epochs 5 --batch-size 2 >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo Training and held-out test evaluation completed.
    echo Report: training_outputs\ravdess_voice\test_metrics.json
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

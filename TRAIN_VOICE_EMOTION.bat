@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo SpeechMind - Voice emotion fine-tuning
echo ==========================================
echo Dataset summary verified:
echo 1,440 RAVDESS speech clips - 24 actors
echo Train: 840 | Validation: 300 | Test: 300
echo.

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Project virtual environment was not found at:
    echo %PYTHON_EXE%
    echo.
    echo Create/restore the project's .venv before training.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo.
    echo PyTorch could not be imported from the project virtual environment.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo WARNING: CUDA is not available. Training will run on CPU and may take many hours.
    echo Do not start this on battery power. Keep the computer plugged in.
    echo.
    choice /C YN /N /M "Continue with CPU training? [Y/N] "
    if errorlevel 2 exit /b 0
)

echo.
echo Starting 5 epochs of speaker-independent training...
echo The test split remains held out until final evaluation.
echo The checkpoint will be saved to checkpoints\voice_emotion_wav2vec2
echo The evaluation report will be saved to training_outputs\ravdess_voice\test_metrics.json
echo.

"%PYTHON_EXE%" "%~dp0scripts\train_voice_emotion.py" --epochs 5 --batch-size 2
set "TRAIN_EXIT=%ERRORLEVEL%"

echo.
if "%TRAIN_EXIT%"=="0" (
    echo Training and held-out test evaluation completed.
) else (
    echo Training did not complete. Exit code: %TRAIN_EXIT%
)
echo.
pause
exit /b %TRAIN_EXIT%

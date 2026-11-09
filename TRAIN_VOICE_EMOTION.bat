@echo off
setlocal EnableExtensions
title SpeechMind - CPU WavLM Feature Training
cd /d "%~dp0"

set "LOG_FILE=%~dp0training_launcher.log"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "EXIT_CODE=0"

> "%LOG_FILE%" echo SpeechMind CPU WavLM feature-training launcher
>> "%LOG_FILE%" echo Started: %DATE% %TIME%
>> "%LOG_FILE%" echo Project folder: %CD%

echo ==========================================================
echo SPEECHMIND - CPU-FRIENDLY WAVLM TRAINING
echo ==========================================================
echo This PC has a Ryzen 5 3500, 16 GB RAM and a GT 730.
echo The GT 730 is not used for modern WavLM CUDA training.
echo This workflow extracts frozen WavLM embeddings on CPU,
echo caches them, then trains a small class-balanced SVM.
echo This is NOT full WavLM fine-tuning.
echo.
echo First run downloads microsoft/wavlm-base-plus.
echo Feature extraction can take a while, but it is cached
echo so later classifier experiments are much faster.
echo.

if not exist "%PYTHON_EXE%" (
    echo ERROR: .venv Python not found. Run SETUP_SECOND_PC.bat first.
    set "EXIT_CODE=2"
    goto FINISH
)

if not exist "%~dp0data\ravdess" (
    echo ERROR: Dataset folder data\ravdess is missing.
    echo Copy Actor_01 through Actor_24 into that folder first.
    set "EXIT_CODE=3"
    goto FINISH
)

echo Checking dataset and PyTorch...
"%PYTHON_EXE%" -c "import torch,sklearn,transformers,soundfile,librosa; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('CPU threads:',torch.get_num_threads())" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo ERROR: A required package import failed. See training_launcher.log.
    set "EXIT_CODE=4"
    goto FINISH
)

echo.
echo Starting CPU feature extraction and classifier selection...
echo Do not close this window. Progress is written to the log.
echo.
>> "%LOG_FILE%" echo Starting CPU WavLM feature training at %DATE% %TIME%
"%PYTHON_EXE%" "%~dp0scripts\train_voice_emotion.py" --epochs 1 >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo CPU WavLM feature training and held-out evaluation completed.
    echo Report: training_outputs\wavlm_cpu\test_metrics.json
    echo Checkpoint: checkpoints\voice_emotion_wavlm_cpu
    >> "%LOG_FILE%" echo Training completed successfully.
) else (
    echo.
    echo Training failed with exit code %EXIT_CODE%.
    echo Read training_launcher.log for the complete error.
    >> "%LOG_FILE%" echo Training failed with exit code %EXIT_CODE%.
)

:FINISH
echo.
echo Exit code: %EXIT_CODE%
echo Log: %LOG_FILE%
echo Press any key to close.
pause >nul
exit /b %EXIT_CODE%

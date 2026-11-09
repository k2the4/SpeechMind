@echo off
setlocal EnableExtensions
title SpeechMind - Multi-Corpus CPU WavLM Training
cd /d "%~dp0"
set "LOG_FILE=%~dp0training_multicorpus.log"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "EXIT_CODE=0"
> "%LOG_FILE%" echo SpeechMind multi-corpus CPU WavLM training
>> "%LOG_FILE%" echo Started: %DATE% %TIME%
>> "%LOG_FILE%" echo Project folder: %CD%
echo ==========================================================
echo SPEECHMIND - MULTI-CORPUS CPU TRAINING
echo ==========================================================
echo Uses RAVDESS plus CREMA-D when CREMA-D is installed.
echo Frozen WavLM top-four-layer features with richer pooling.
echo CPU features are cached for repeat experiments.
echo Results are reported combined and separately by dataset.
echo.
if not exist "%PYTHON_EXE%" (
 echo ERROR: .venv Python not found. Run SETUP_SECOND_PC.bat first.
 set "EXIT_CODE=2"
 goto FINISH
)
if not exist "%~dp0data\ravdess" (
 echo ERROR: data\ravdess is missing.
 set "EXIT_CODE=3"
 goto FINISH
)
if not exist "%~dp0data\crema_d\AudioWAV" (
 echo ERROR: CREMA-D is required for this expanded experiment.
 echo Complete the CREMA-D access registration, install Git LFS,
 echo and run DOWNLOAD_CREMA_D.bat first.
 set "EXIT_CODE=4"
 goto FINISH
)
echo CREMA-D detected. Both corpora will be used.
echo.
echo Starting feature extraction and classifier evaluation.
echo Do not close this window. Detailed progress is written to the log.
>> "%LOG_FILE%" echo Starting training at %DATE% %TIME%
"%PYTHON_EXE%" -u "%~dp0scripts\train_voice_emotion.py" --require-crema-d >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
 echo Training and held-out evaluation completed.
 echo Report: training_outputs\wavlm_multicorpus\test_metrics.json
 echo Checkpoint: checkpoints\voice_emotion_multicorpus_cpu
) else (
 echo Training failed with exit code %EXIT_CODE%.
 echo Read training_multicorpus.log.
)
goto FINISH
:CANCELLED
echo Training cancelled. Install CREMA-D first for expanded experiment.
set "EXIT_CODE=0"
:FINISH
echo.
echo Exit code: %EXIT_CODE%
echo Log: %LOG_FILE%
pause
exit /b %EXIT_CODE%

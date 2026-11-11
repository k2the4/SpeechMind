@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "CHECKPOINT=%~dp0checkpoints\text_emotion_muril"

if not exist "%PYTHON_EXE%" (
    echo SpeechMind main environment not found. Run SETUP_SECOND_PC.bat first.
    pause
    exit /b 1
)
if not exist "%CHECKPOINT%\config.json" (
    echo Trained MuRIL checkpoint config is missing.
    echo Run TRAIN_TEXT_EMOTION.bat first.
    pause
    exit /b 1
)
if not exist "%CHECKPOINT%\speechmind_metadata.json" (
    echo Checkpoint metadata is missing.
    pause
    exit /b 1
)
if not exist "%CHECKPOINT%\model.safetensors" if not exist "%CHECKPOINT%\pytorch_model.bin" (
    echo Model weights are missing from the checkpoint.
    pause
    exit /b 1
)

echo Checking the locally trained Hindi/Hinglish text-emotion model...
"%PYTHON_EXE%" -m scripts.check_text_emotion
if errorlevel 1 (
    echo.
    echo Text-emotion checkpoint check FAILED.
    pause
    exit /b 1
)
echo.
echo Text-emotion checkpoint check PASSED.
pause
exit /b 0

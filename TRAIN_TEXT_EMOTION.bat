@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo SpeechMind main environment not found. Run SETUP_SECOND_PC.bat first.
    pause
    exit /b 1
)

if not exist "data\text_emotion\prepared\brighter_hi_train.jsonl" (
    echo Prepared text-emotion data not found. Starting data preparation...
    call "%~dp0PREPARE_TEXT_EMOTION_DATA.bat"
    if errorlevel 1 exit /b 1
)

echo ==========================================================
echo SpeechMind - Train Hindi/Hinglish text-emotion model
echo ==========================================================
echo MuRIL multi-label fine-tuning on CPU/GPU as available.
echo BRIGHTER Hindi labels are human annotated; HinEmo GPT labels are
echo down-weighted and masked for emotions that dataset does not provide.
echo This can take time on CPU. Do not close the console during training.
echo.

"%PYTHON_EXE%" "%~dp0scripts\train_text_emotion_multilabel.py" --max-hinemo-train 5000 --epochs 2 --batch-size 4 --gradient-accumulation 4 --trainable-layers 2
if errorlevel 1 (
    echo Text-emotion training failed. Review the error above.
    pause
    exit /b 1
)

echo.
echo Training completed. Restart SpeechMind to load the new multilingual checkpoint.
pause
exit /b 0
